from langchain_core.documents import Document

chunk = Document(
    page_content=('. 진단 당시의 한국표준질병․사인분류에 따라 이 약관에서 보 장하는 질병(상병)에 대한 보험금 지급여부가 판단된 경우, 이후 '
 '한국표준질병․사인분류 개정으로 질병(상병)분류가 변 경되더라도 이 약관에서 보장하는 질병(상병) 해당 여부를 다시 판단하지 '
 "않습니다.</td></tr></tbody></table><p id='4' data-category='paragraph' "
 "style='font-size:18px'>- 54 -</p><table id='5'"),
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
 'indexing': {'chunk_id': 'chunk_000009',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
