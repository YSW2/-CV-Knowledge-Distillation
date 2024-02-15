plane_cifar10_book = {
	'2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
	'4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
	'6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
	'8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
		  'Conv128', 'Conv128','MaxPool', 'FC64', 'FC10'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC128' ,'FC10'],
}


plane_cifar100_book = {
	'2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
	'4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
	'6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool','Conv128', 'Conv128' ,'FC100'],
	'8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		  'Conv256', 'Conv256','MaxPool', 'FC64', 'FC100'],
	'10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
		   'Conv256', 'Conv256', 'Conv256', 'Conv256' , 'MaxPool', 'FC512', 'FC100'],
}