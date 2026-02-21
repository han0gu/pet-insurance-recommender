from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나\n'
 '- 진단서 위·변조 또는 청약일 이전에 암 또는 사람면역결핍바이러스병(HIV) 감염\n'
 '- 의 진단 확정을 받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성립되었 특별\n'
 '- 음을 회사가 증명하는 경우에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월\n'
 '- 약\n'
 '- 이내)에 계약을 취소할 수 있습니다.\n'
 '- 관\n'
 '- \uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게\n'
 '- 돌려드립니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
