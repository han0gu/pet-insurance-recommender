from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="4">간질영향 호흡기질환</td><td>대상이 되는 '
 '항목</td><td>분류번호</td></tr><tr><td>성인호흡곤란증후군</td><td>J80 '
 '보</td></tr><tr><td>폐부종</td><td>J81</td></tr><tr><td>달리 분류되지 않은 '
 '폐호산구증가</td><td>통약 J82</td></tr><tr><td rowspan="2">하기도 화농·</td><td>기타 간질성 '
 '폐질환 J84</td><td>관</td></tr><tr><td>폐 및 종격의'),
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
 'indexing': {'chunk_id': 'chunk_001759',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
