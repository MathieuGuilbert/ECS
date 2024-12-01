def verifyMustLink(a:int,b:int,Cid:list):
	'''Verify the satisfaction of a Must-Link constraint.
	
	PARAMETERS
	--------
	a: id of an instance.
	b: id of another instance.
	Cid: list of the ids of the instances in a particular cluster.

	RESULTS
	---------
	Boolean: True if the constraint is satisfied, False otherwise.
	'''
	if (a in Cid):
		if (b in Cid):
			return True
		else:
			return False
	else:
		if (b in Cid):
			return False
		else:
			return True

def verifyCannotLink(a:int,b:int,Cid:list):
	'''Verify the satisfaction of a Cannot-Link constraint.
	
	PARAMETERS
	--------
	a: id of an instance.
	b: id of another instance.
	Cid: list of the ids of the instances in a particular cluster.

	RESULTS
	---------
	Boolean: True if the constraint is satisfied, False otherwise.
	'''
	if (a in Cid):
		if (b in Cid):
			return False
		else:
			return True
	else:
		return True