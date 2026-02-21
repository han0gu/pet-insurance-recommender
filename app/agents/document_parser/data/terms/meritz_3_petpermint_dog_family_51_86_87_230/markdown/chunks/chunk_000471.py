from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 흡인(吸引): 주사기 등으로 빨아들이는 것\n'
 '- - 천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거\n'
 '- 나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물\n'
 '병원에서 행한 것에 한합니다.# 제4조(입원의 정의와 장소)이 특별약관에 있어서 「입원」이라 함은 수의사가 상해 또\n'
 '는 질병의 치료가 필요하다고 인정한 경우로서, 자택 등에\n'
 '서의 치료가 곤란하여 동물병원에 입실하여 수의사의 관리'),
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
