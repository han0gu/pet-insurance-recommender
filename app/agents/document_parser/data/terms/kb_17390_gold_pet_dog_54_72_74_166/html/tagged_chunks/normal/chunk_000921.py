from langchain_core.documents import Document

chunk = Document(
    page_content=("KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='104' style='font-size:14px'>내용에 "
 '상응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기<br>손해보험에 한하며 이하 "반려동물보험 상품"이라 합니다)으로 '
 '가입을 할 수 있<br>으며, 회사는 이를 거절할 수 없습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000921',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
