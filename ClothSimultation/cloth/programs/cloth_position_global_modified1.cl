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
	int idx = get_global_id(0) + get_global_size(0) * get_global_id(1);

	float4 r;
	float3 v = vel_in[idx].xyz;
	float3 force = Gravity * ParticleMass;
	
	if (get_global_id(1) < get_global_size(1) - 1)
	{
		r = pos_in[idx + get_global_size(0)] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthVert);
	}

	if (get_global_id(1) > 0)
	{
		r = pos_in[idx - get_global_size(0)] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthVert);
	}

	if (get_global_id(0) > 0)
	{
		r = pos_in[idx - 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthHoriz);
	}

	if (get_global_id(0) < get_global_size(0) - 1)
	{
		r = pos_in[idx + 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthHoriz);
	}

	// Diagonals

	if (get_global_id(0) > 0 && get_global_id(1) < get_global_size(1) - 1)
	{
		r = pos_in[idx + get_global_size(0) - 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) < get_global_size(0) - 1 && get_global_id(1) < get_global_size(1) - 1)
	{
		r = pos_in[idx + get_global_size(0) + 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) > 0 && get_global_id(1) > 0)
	{
		r = pos_in[idx - get_global_size(0) - 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}

	if (get_global_id(0) < get_global_size(0) - 1 && get_global_id(1) > 0)
	{
		r = pos_in[idx - get_global_size(0) + 1] - pos_in[idx];
		force += normalize(r).xyz * SpringK * (length(r) - RestLengthDiag);
	}


	force += -DampingConst * v;
	float3 a = force * ParticleInvMass;
	
	// Position of Particles

	pos_out[idx] = pos_in[idx] + vel_in[idx] * DeltaT + (float4)( a * DeltaT * DeltaT / 2, 0.0);
	//Method 2 : 수업시간에 설명한 방법

	//Velocity of Particles                                                                

	vel_out[idx] = vel_in[idx] + (float4)(a * DeltaT, 0.0);
	//Method 2 : 수업시간에 설명한 방법

	if (get_global_id(1) == get_global_size(1) - 1 &&
		(get_global_id(0) == 0 ||
			get_global_id(0) == get_global_size(0) / 4 ||
			get_global_id(0) == get_global_size(0) * 2 / 4 ||
			get_global_id(0) == get_global_size(0) * 3 / 4 ||
			get_global_id(0) == get_global_size(0) - 1))
	{
		pos_out[idx] = pos_in[idx];
		vel_out[idx] = (float4)(0, 0, 0, 0);
	}
}