from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항에 따라 개별계약으로 전환시에는 전환후 피보험자의 보험기간은 이 계약의 남은\n'
 '- 기간으로 하고, 이로 인하여 발생하는 추가 또는 환급되는 보험료는 보험료 및 해약환\n'
 '- 급금 산출방법서에서 정한 바에 따라 일단위로 계산하여 받거나 돌려 드립니다.\n'
 '# 제6조(보험증권의 발급)- ① 회사는 보험계약자에게 보험증권을 드려야 하고, 그 약관의 주요한 내용을 알려드립니다.\n'
 '- ② 보험계약자의 요청이 있을 경우, 개별 피보험자에게는 가입증명서를 발급하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
