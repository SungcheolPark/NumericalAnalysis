__kernel
void cloth_position(
	__global float4* pos_in, __global float4* pos_out,
	__global float4* vel_in, __global float4* vel_out,
	__local float4* local_data,
	float3 Gravity,
	float ParticleMass,
	float ParticleInvMass,
	float SpringK,
	float RestLengthHoriz,
	float RestLengthVert,
	float RestLengthDiag,
	float DeltaT,
	float DampingConst) {
	int global_idx = get_global_id(0) + get_global_size(0) * get_global_id(1);
	int local_idx = (get_local_id(0) + 1) + (get_local_size(0) + 2) * (get_local_id(1) + 1);

	float4 r;
	float3 v = r.xyz;
	float3 force = Gravity * ParticleMass;

	local_data[local_idx] = pos_in[global_idx];  // position copy

	if (get_local_id(0) == 0) {  // 위쪽
		local_data[get_local_id(0) + (get_local_size(0) + 2) * (get_local_id(1) + 1)] = pos_in[(get_global_id(0) - 1) + get_global_size(0) * get_global_id(1)];
	}
	if (get_local_id(1) == 0) {   // 왼쪽
		local_data[(get_local_id(0) + 1) + (get_local_size(0) + 2) * get_local_id(1)] = pos_in[get_global_id(0) + get_global_size(0) * (get_global_id(1) - 1)];
	}
	if (get_local_id(0) == (get_local_size(0) - 1)) {  // 아래쪽
		local_data[(get_local_id(0) + 2) + (get_local_size(0) + 2) * (get_local_id(1) + 1)] = pos_in[(get_global_id(0) + 1) + get_global_size(0)* get_global_id(1)];
	}
	if (get_local_id(1) == (get_local_size(1) - 1)) {  // 오른쪽
		local_data[(get_local_id(0) + 1) + (get_local_size(0) + 2) * (get_local_id(1) + 2)] = pos_in[get_global_id(0) + get_global_size(0)* (get_global_id(1) + 1)];
	}


	/* corner */
	if (get_local_id(0) == 0 && get_local_id(1) == 0) { // 왼쪽 위 귀퉁이
		local_data[0] = pos_in[(get_global_id(0) - 1) + get_global_size(0)* (get_global_id(1) - 1)];
	}
	else if (get_local_id(0) == 0 && get_local_id(1) == (get_local_size(1) - 1)) {  // 오른쪽 위 귀퉁이
		local_data[get_local_id(0) + (get_local_size(0) + 2) * (get_local_id(1) + 2)] = pos_in[(get_global_id(0) - 1) + get_global_size(0)* (get_global_id(1) + 1)];
	}
	else if (get_local_id(0) == (get_local_size(0) - 1) && get_local_id(1) == 0) {  // 왼쪽 아래 귀퉁이
		local_data[(get_local_id(0) + 2) + (get_local_size(0) + 2) * get_local_id(1)] = pos_in[(get_global_id(0) + 1) + get_global_size(0)* (get_global_id(1) - 1)];
	}
	else if (get_local_id(0) == (get_local_size(0) - 1) && get_local_id(1) == (get_local_size(1) - 1)) {  // 오른쪽 아래 귀퉁이
		local_data[(get_local_id(0) + 2) + (get_local_size(0) + 2) * (get_local_id(1) + 2)] = pos_in[(get_global_id(0) + 1) + get_global_size(0)* (get_global_id(1) + 1)];
	}

	barrier(CLK_LOCAL_MEM_FENCE); // local memory fence

								  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	if (get_global_id(1) < get_global_size(1) - 1)
	{
		r = local_data[local_idx + (get_local_size(0) + 2)] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthVert);
	}

	if (get_global_id(1) > 0)
	{
		r = local_data[local_idx - (get_local_size(0) + 2)] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthVert);
	}

	if (get_global_id(0) > 0)
	{
		r = local_data[local_idx - 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthHoriz);
	}

	if (get_global_id(0) < get_global_size(0) - 1)
	{
		r = local_data[local_idx + 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthHoriz);
	}

	// Diagonals

	if (get_global_id(0) > 0 && get_global_id(1) < get_global_size(1) - 1)
	{
		r = local_data[local_idx + (get_local_size(0) + 2) - 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) < get_global_size(0) - 1 && get_global_id(1) < get_global_size(1) - 1)
	{
		r = local_data[local_idx + (get_local_size(0) + 2) + 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) > 0 && get_global_id(1) > 0)
	{
		r = local_data[local_idx - (get_local_size(0) + 2) - 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) < get_global_size(0) - 1 && get_global_id(1) > 0)
	{
		r = local_data[local_idx - (get_local_size(0) + 2) + 1] - local_data[local_idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	force += -DampingConst * v;

	float3 a = force * ParticleInvMass;

	// Position of Particles
	
	pos_out[global_idx] = local_data[local_idx] + vel_in[global_idx] * DeltaT + (float4)((float)(0.5) * a * DeltaT * DeltaT, 0.0);               
	//Method 2 : 수업시간에 설명한 방법
	
	//Velocity of Particles           
	
	vel_out[global_idx] = vel_in[global_idx] + (float4)(a * DeltaT, 0.0);
	//Method 2 : 수업시간에 설명한 방법

	if (get_global_id(1) == get_global_size(1) - 1 &&
		(get_global_id(0) == 0 ||
			get_global_id(0) == get_global_size(0) / 4 ||
			get_global_id(0) == get_global_size(0) * 2 / 4 ||
			get_global_id(0) == get_global_size(0) * 3 / 4 ||
			get_global_id(0) == get_global_size(0) - 1))
	{
		pos_out[global_idx] = pos_in[global_idx];
		vel_out[global_idx] = (float4)(0, 0, 0, 0);
	}

}