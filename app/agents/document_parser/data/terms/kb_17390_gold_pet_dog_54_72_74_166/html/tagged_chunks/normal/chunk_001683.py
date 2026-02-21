from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="3">옷 입고</td><td>1) 상·하의 의복 착탈시 다른 사람의 계속적인 도움이 필요한 '
 '상태</td><td>10%</td></tr><tr><td>2) 상·하의 의복 착탈시 부분적으로 다른 사람의 도움이 필요한 상태 또는 상의 '
 '또는 하의중 하나만 혼자서 착 벗기 탈의가 가능한 상태</td><td>5%</td></tr><tr><td>3) 상·하의 의복착탈시 혼자서 '
 '가능하나 미세동작(단추 잠그고 풀기, 지퍼 올리고 내리기, 끈 묶고 풀기 등) 이 필요한 마무리는 타인의 도움이 필요한'),
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
 'indexing': {'chunk_id': 'chunk_001683',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
