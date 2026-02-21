from langchain_core.documents import Document

chunk = Document(
    page_content=("및 해약환급금 산출방법서」에<br>따라 산출합니다.</p><br><p id='19' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제6항에 따라 보험계약이 연장된 경우 계약자는 그 최초<br>연장된 날로부터 90일 "
 '이내에 그 계약을 취소할 수 있으며,<br>계약자가 연장된 보험계약을 취소하는 경우 회사는 최초연<br>장된 날 이후 계약자가 납입한 '
 '보험료 전액을 환급합니다.<br>\uf000 제6항에 따라 보험계약이 연장된 경우 보험계약의 연장<br>일은 회사가 계약자의 재가입의사를 '
 '확인한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000370',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
