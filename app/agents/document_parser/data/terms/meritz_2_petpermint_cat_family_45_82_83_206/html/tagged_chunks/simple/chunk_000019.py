from langchain_core.documents import Document

chunk = Document(
    page_content=("회<br>사의 사업경비를 위한 부가보험료로 구성됩니다.</p><footer id='27' "
 "style='font-size:14px'>49</footer><h1 id='28' style='font-size:16px'>보험료 = "
 '보장보험료 + 적립보험료<br>보장보험료 = 위험보험료 + 부가보험료<br>적립보험료 = 적립부분 순보험료 + 부가보험료</h1><p '
 "id='29' data-category='paragraph' style='font-size:20px'>제2관 보험금의 지급</p><p "
 "id='30'"),
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
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
