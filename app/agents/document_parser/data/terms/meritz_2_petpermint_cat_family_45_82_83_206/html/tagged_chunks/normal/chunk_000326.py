from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 중대한 과실로 제1항<br>각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요<br>율이 변경전 요율보다 높을 때에는 회사는 '
 '그 변경사실을<br>안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에<br>따라 보장됨을 통보하고 이에 따라 보험금을 '
 "지급합니다.</p><br><h1 id='68' style='font-size:20px'>【중대한 과실】</h1><br><p id='69' "
 "data-category='paragraph' style='font-size:20px'>주의의무의 위반이 현저한 과실, 즉 현저한"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000326',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
