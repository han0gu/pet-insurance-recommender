from langchain_core.documents import Document

chunk = Document(
    page_content=("용어의 정의 】</h1><br><p id='36' data-category='list' style='font-size:14px'>- "
 '절단(切斷): 특정부위를 잘라 내는 것<br>- 절제(切除): 특정부위를 잘라 없애는 것<br>- 흡인(吸引): 주사기 등으로 빨아들이는 '
 "것<br>- 천자(穿刺): 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것</p><h1 id='37' "
 "style='font-size:14px'>제5조(보험금을 지급하지 않는 사유)</h1><br><p id='38'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
