from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>지정한 질</p><br><p id='120' data-category='paragraph' "
 'style=\'font-size:14px\'>병(이하"특정질병"이라 합니다)(【별표17】(반려동물(강아지) 특정 질병 분류표))<br>을 '
 '직접적인 원인으로 보험계약에서 정한 보험금지급사유가 발생한 경우에는 회사<br>는 보험금을 지급하지 않습니다.</p><br><p '
 "id='121' data-category='paragraph' style='font-size:16px'>- 138 -</p><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001447',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
