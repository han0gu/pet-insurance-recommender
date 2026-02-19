from langchain_core.documents import Document

chunk = Document(
    page_content=('목적의 수술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검, 복강경검사 등)은 제외합니 다.\n'
 '【 용어의 정의 】\n'
 '- 절단(切斷): 특정부위를 잘라 내는 것 - 절제(切除): 특정부위를 잘라 없애는 것 - 흡인(吸引): 주사기 등으로 빨아들이는 것 - '
 '천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것\n'
 '제5조(보험금을 지급하지 않는 사유)\n'
 '① 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 4},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
