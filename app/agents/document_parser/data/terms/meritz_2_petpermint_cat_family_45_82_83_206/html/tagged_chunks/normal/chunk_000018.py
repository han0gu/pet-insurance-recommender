from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>【보험료】</h1><br><p id='26' data-category='paragraph' "
 "style='font-size:16px'>보험료는 계약자가 계약에 따라 회사에게 지급하여야 하<br>는 요금을 말하며, "
 '보험료는「보장보험료」와「적립보험<br>료」로 구성되어 있습니다.<br>또한, 보험료는 보험금 지급을 위한 위험보험료, 회사가<br>적립한 '
 "금액을 돌려주기 위한 적립부분 순보험료 및 회<br>사의 사업경비를 위한 부가보험료로 구성됩니다.</p><footer id='27'"),
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
 'indexing': {'chunk_id': 'chunk_000018',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
