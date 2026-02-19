from langchain_core.documents import Document

chunk = Document(
    page_content=('【용어의 정의】\n'
 '- 절단(切斷): 특정부위를 잘라 내는 것 - 절제(切除): 특정부위를 잘라 없애는 것 - 흡인(吸引): 주사기 등으로 빨아들이는 것 - '
 '천자(穿刺): 바늘 또는 관을 꽂아 체액ㆍ조직을 뽑아내 거나 약물을 주입하는 것\n'
 '\uf000 제1항의「수술」은 자택 등에서의 치료가 곤란하여 동물 병원에서 행한 것에 한합니다.\n'
 '제4조(특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000325',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
