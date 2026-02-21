from langchain_core.documents import Document

chunk = Document(
    page_content=("효력)</p><br><p id='41' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 일반금융소비자에게 청약을 권유하거나 일반금융<br>소비자가 설명을 요청하는 "
 '경우 보험상품에 관한 중요한 사<br>항을 계약자가 이해할 수 있도록 설명하고 계약자가 이해하<br>였음을 서명(「전자서명법」 제2조 '
 '제2호에 따른 전자서명<br>을 포함), 기명날인 또는 녹취 등을 통해 확인받아야 하며,<br>설명서를 제공하여야 '
 '합니다.<br>\uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000246',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
