from langchain_core.documents import Document

chunk = Document(
    page_content=("효과)</h1><br><p id='48' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생<br>여부에 관계없이 이 "
 "계약을 해지할 수 있습니다.</p><br><p id='49' data-category='list' "
 "style='font-size:20px'>① 계약자 또는 피보험자가 고의 또는 중대한 과실로 제<br>15조(계약 전 알릴 의무)를 "
 '위반하고 그 의무가 중요<br>한 사항에 해당하는 경우<br>② 뚜렷한 위험의 증가와 관련된'),
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
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
