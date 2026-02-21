from langchain_core.documents import Document

chunk = Document(
    page_content=('| 804 | 안과질환 | 유루증 표 |\n'
 '| 804 | 안과질환 | 전안방출혈 |\n'
 '| 804 | 안과질환 | 제3안검염 |\n'
 '| 804 | 안과질환 | 제3안검의 탈출 |\n'
 '| 804 | 안과질환 | 첩모질환 (난생, 중생, 이소성) 법 |\n'
 '| 804 | 안과질환 | 포도막염 ㆍ |\n'
 '| 804 | 안과질환 | 기타 선천성 안과질환 규정 |\n'
 '| 804 | 안과질환 | 기타 안과 질환 |\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 165- 165 -|  | 코드 특정 | 질병 세부 질병명 |\n'
 '| --- | --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_001040',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
