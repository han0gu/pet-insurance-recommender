from langchain_core.documents import Document

chunk = Document(
    page_content=('내 / 외번증</td></tr><tr><td>안검염</td></tr><tr><td>안방수 흐림</td></tr><tr><td>안와 형성 '
 '부전 별</td></tr><tr><td>유루증 '
 '표</td></tr><tr><td>전안방출혈</td></tr><tr><td>제3안검염</td></tr><tr><td>제3안검의 '
 '탈출</td></tr><tr><td>첩모질환 (난생, 중생, 이소성) 법</td></tr><tr><td>포도막염 '
 'ㆍ</td></tr><tr><td>기타 선천성 안과질환 규정</td></tr><tr><td>기타 안과'),
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
 'indexing': {'chunk_id': 'chunk_001815',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
