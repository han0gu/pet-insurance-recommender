from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(손해의 발생과 통지)\n'
 '회사의 요구에 따르기 위하여 지출한 비용 반\n'
 '\uf000 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을\n'
 '회사에 알려야 합니다. 려동\n'
 '제5조(보상하지 않는 손해)\n'
 '1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고 물\n'
 '\uf000 회사는 아래의 사유로 인한 손해는 보상하여 드리지 않습니다.\n'
 '상황 및 이들 사항의 증인이 있을 경우 그 주소와 성명\n'
 '1. 계약자, 피보험자 또는 이들의 법정대리인의 고의로 생긴 손해에 대한 배상책임'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000670',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
