from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험기간 중에 공시이율이 변경되는 경우에 변경된 시점 이후에는 변 특별<br>경된 이율을 적용합니다.</p><br><p '
 'id=\'80\' data-category=\'list\' style=\'font-size:14px\'>약<br>1. "보장성-1701 '
 '공시이율"은 매월 마지막날 회사가 정한 이율로 하며, 다음달 1일 관<br>부터 마지막날까지 1개월간 확정 적용합니다.<br>2'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
