phones = [
	'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',
	'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
	'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
	'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
	'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
	'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
	'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
	'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
	'v', 'w', 'y', 'z', 'zh'
]

phonesToInt = dict(zip(phones, range(len(phones))))
intToPhones = dict(zip(range(len(phones)), phones))


phonesMapping = {
	'aa': 'aa', 'ao': 'aa',
	'ah': 'ah', 'ax': 'ah', 'ax-h': 'ah',
	'er': 'er', 'axr': 'er',
	'hh': 'hh', 'hv': 'hh',
	'ih': 'ih', 'ix': 'ih',
	'l': 'l', 'el': 'l',
	'm': 'm', 'em': 'm',
	'n': 'n', 'en': 'n', 'nx': 'n',
	'ng': 'ng', 'eng': 'ng',
	'sh': 'sh', 'zh': 'sh',
	'uw': 'uw', 'ux': 'uw',
	'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil', 'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
	'q': '-'
}


intToReducedPhones = {}
for phone in phones:
	if phone == 'q':
		continue

	if phone in phonesMapping:
		intToReducedPhones[phonesToInt[phone]] = phonesMapping[phone]
	else:
		intToReducedPhones[phonesToInt[phone]] = phone
