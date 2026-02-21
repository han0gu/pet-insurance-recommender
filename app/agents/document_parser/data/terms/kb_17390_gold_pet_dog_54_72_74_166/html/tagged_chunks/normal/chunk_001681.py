from langchain_core.documents import Document

chunk = Document(
    page_content=('화장실에 가는 일, 배변, 배뇨는 독립적으로 가능하나 대소변후 뒤처리에 있어 다른 사람의 도움이 필요한 '
 '상태</td><td>10%</td></tr><tr><td>4) 빈번하고 불규칙한 배변으로 인해 2시간 이상 계속되 는 업무를 수행하는 '
 '것이 어려운 상태, 또는 배변, 배 뇨는 독립적으로 가능하나 요실금, 변실금이 있는 때</td><td>5%</td></tr><tr><td '
 'rowspan="3"></td><td>1) 세안, 양치, 샤워, 목욕 등 모든 개인위생 관리시 타 인의 지속적인 도움이 필요한'),
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
 'indexing': {'chunk_id': 'chunk_001681',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
