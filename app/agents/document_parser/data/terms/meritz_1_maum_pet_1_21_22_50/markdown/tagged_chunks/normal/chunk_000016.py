from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하는 것을 말합니다. 단, 흡인, 천자 등의 조치, 신경(神經)차단(NERVE BLOCK), 미용성형\n'
 '- 3 -목적의 수술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검, 복강경검사 등)은 제외합니\n'
 '다.# 【 용어의 정의 】- - 절단(切斷): 특정부위를 잘라 내는 것\n'
 '- - 절제(切除): 특정부위를 잘라 없애는 것\n'
 '- - 흡인(吸引): 주사기 등으로 빨아들이는 것\n'
 '- - 천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
