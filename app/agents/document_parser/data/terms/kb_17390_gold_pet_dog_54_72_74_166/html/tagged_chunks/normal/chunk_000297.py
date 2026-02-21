from langchain_core.documents import Document

chunk = Document(
    page_content=(". 법<br>ㆍ</p><br><p id='129' data-category='paragraph' "
 "style='font-size:16px'>제40조(지정대리청구인의 변경지정)</p><br><p id='130' "
 "data-category='paragraph' style='font-size:16px'>계약자는 계약체결 이후 다음의 "
 "서류를</p><br><p id='131' data-category='paragraph' style='font-size:16px'>습니다"),
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
 'indexing': {'chunk_id': 'chunk_000297',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
