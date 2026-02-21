from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 흡인(吸引): 주사기 등으로 빨아들이는 것\n'
 '- - 천자(穿刺): 바늘 또는 관을 꽂아 체액ㆍ조직을 뽑아내\n'
 '- 거나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물\n'
 '병원에서 행한 것에 한합니다.# 제4조(MRI,CT 및 내시경처치의 정의)\uf000 이 특별약관에 있어서 MRI,CT 및 내시경처치라 '
 '함은 자\n'
 '기공명영상(MRI), 전산화단층촬영(CT) 및 내시경처치를 말\n'
 '합니다.\n'
 '\uf000 제1항의 자기공명영상(MRI)이라 함은 제1조(보험금의 지\n'
 '급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
