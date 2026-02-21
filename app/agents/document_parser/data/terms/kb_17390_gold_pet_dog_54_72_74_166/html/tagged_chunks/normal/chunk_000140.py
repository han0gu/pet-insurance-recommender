from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자가 제1회 보험료<br>를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하<br>며 '
 "이자를 더하여 지급하지 않습니다.</p><br><p id='164' data-category='paragraph' "
 "style='font-size:14px'>별</p><br><p id='165' data-category='paragraph' "
 "style='font-size:14px'>표</p><p id='166' data-category='paragraph' "
 "style='font-size:16px'>KB"),
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
 'indexing': {'chunk_id': 'chunk_000140',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
