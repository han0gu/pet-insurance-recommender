from langchain_core.documents import Document

chunk = Document(
    page_content=('조치, 신경(神經)차단(NERVE BLOCK), 미용성형 목적의 수\n'
 '술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검, 복\n'
 '강경검사 등)은 제외합니다.# 【용어의 정의】- - 절단(切斷): 특정부위를 잘라 내는 것\n'
 '- - 절제(切除): 특정부위를 잘라 없애는 것\n'
 '- - 흡인(吸引): 주사기 등으로 빨아들이는 것\n'
 '- - 천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거\n'
 '- 나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000296',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
