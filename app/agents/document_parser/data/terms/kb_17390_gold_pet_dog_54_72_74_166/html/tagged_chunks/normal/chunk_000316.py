from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>일반금융소비자가 설명을 요청하</p><br><p id='166' "
 "data-category='paragraph' style='font-size:14px'>는 경우 보험상품에 관한 중요한 사항을 계약자가 "
 '이해할 수 있도록 설명하고 계<br>약자가 이해하였음을 서명(전자서명법 제2조 제2호에 따른 전자서명을 포함), 기<br>명날인 또는 '
 '녹취 등을 통해 확인받아야 하며, 설명서를 제공하여야 합니다.<br>\uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제공 '
 '사실에 관하여'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000316',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
