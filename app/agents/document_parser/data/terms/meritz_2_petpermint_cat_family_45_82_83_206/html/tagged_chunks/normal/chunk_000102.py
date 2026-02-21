from langchain_core.documents import Document

chunk = Document(
    page_content=('그 의무가 중요<br>한 사항에 해당하는 경우<br>② 뚜렷한 위험의 증가와 관련된 제16조(상해보험계약 후<br>알릴 의무) 제1항에서 '
 '정한 계약 후 알릴 의무를 계약<br>자 또는 피보험자의 고의 또는 중대한 과실로 이행하<br>지 않았을 때</p><br><p '
 "id='50' data-category='paragraph' style='font-size:20px'>\uf000 제1항 제1호의 "
 "경우에도 불구하고 다음 중 하나에 해당<br>하는 경우에는 회사는 계약을 해지할 수 없습니다.</p><br><p id='51'"),
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
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
