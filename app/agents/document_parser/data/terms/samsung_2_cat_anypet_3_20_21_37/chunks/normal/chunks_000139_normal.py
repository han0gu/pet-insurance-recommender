from langchain_core.documents import Document

chunk = Document(
    page_content=('① 피보험자가 퇴직 등의 사유로 인하여 피보험단체에서 탈퇴하는 경우 피보험자가 보험료의 일부를 부담한 경우에 한하여 탈퇴일로부터 1개월 '
 '이내에 계약자 또는 피보험자는 회사의 승낙을 얻어 개 별계약으로 전환할 수 있으며, 이 경우 피보험자는 개별계약의 계약자가 됩니다. ② '
 '제1항에 따라 개별계약으로 전환시에는 전환후 피보험자의 보험기간은 이 계약의 남은 기간으로 하고, 이로 인하여 발생하는 추가 또는 '
 '환급되는 보험료는 보험료 및 해약환급금 산출방법서에서 정한 바에 따라 일단위로 계산하여 받거나 돌려 드립니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
