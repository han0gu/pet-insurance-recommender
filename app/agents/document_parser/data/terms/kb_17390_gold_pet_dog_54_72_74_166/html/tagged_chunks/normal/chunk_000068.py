from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>규정</p><table id='85' "
 "style='font-size:14px'><thead><tr><td></td></tr></thead><tbody><tr><td>용 어 풀 "
 '이 ∙ 공시이율 전통적인 보험상품에 적용되는 이율이 장기·고정금리이기 때문에 시중금리가 급격하게 변동할 경우 이에 대응하지 못하는 점을 '
 '고려하여, 시중의 지표금리 등에 연동하여 일정기간 마다 변동되는 이율을 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
