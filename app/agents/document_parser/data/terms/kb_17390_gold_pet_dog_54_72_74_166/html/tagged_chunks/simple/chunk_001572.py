from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:14px'>3) ‘빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골), "
 '어깨뼈(견갑골)에 뚜렷한<br>기형이 남은 때’라 함은 방사선 검사로 측정한 각(角) 변형이 20° 이<br>상인 경우를 '
 '말한다.<br>4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관계없이 전체를 일괄<br>하여 하나의 장해로 취급한다'),
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
 'indexing': {'chunk_id': 'chunk_001572',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
