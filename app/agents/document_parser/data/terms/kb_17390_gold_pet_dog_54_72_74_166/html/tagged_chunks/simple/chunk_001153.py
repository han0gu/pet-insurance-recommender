from langchain_core.documents import Document

chunk = Document(
    page_content=('. 벌과금 및 징벌적 손해에 대한 배상책임<br>나. 피보험자가 제12조(손해방지의무) 제1항 제2호의 제3자로부터 손해의 '
 '배<br>11. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한<br>상을 받을 수 있는 그 권리를 '
 '지키거나 행사하기 위하여 지출한 필요 또는<br>상<br>목적으로 이용함으로써 발생한 손해<br>유익하였던 비용 해<br>12. '
 '가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임<br>다'),
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
 'indexing': {'chunk_id': 'chunk_001153',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
