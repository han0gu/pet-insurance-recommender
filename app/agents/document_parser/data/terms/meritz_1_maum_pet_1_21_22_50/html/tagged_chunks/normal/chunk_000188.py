from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:14px'>① 회사는 일반금융소비자에게 청약을 권유하거나 "
 '일반금융소비자가 설명을 요청하는 경우<br>보험상품에 관한 중요한 사항을 계약자가 이해할 수 있도록 설명하고 계약자가 이해하<br>였음을 '
 '서명(⌜전자서명법⌟ 제2조 제2호에 따른 전자서명을 포함), 기명날인 또는 녹<br>취 등을 통해 확인받아야 하며, 설명서를 제공하여야 '
 '합니다.<br>② 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제공 사실에 관하여 계약자와 회사<br>간에 다툼이 있는 경우에는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
