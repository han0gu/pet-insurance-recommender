from langchain_core.documents import Document

chunk = Document(
    page_content=("교부하고 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여</p><br><h1 id='41' "
 "style='font-size:14px'>드립니다.</h1><br><p id='42' data-category='list' "
 "style='font-size:14px'>제14조(보험나이 등)<br>\uf000 이 특별약관에서의 반려동물의 나이는 만나이를 기준으로 "
 '합니다.<br>\uf000 제1항의 만나이는 계약일 현재 반려동물의 실제 만나이를 기준으로 하며, 이후 매<br>년 계약 해당일에 나이가 '
 '증가하는 것으로 합니다.<br>\uf000 청약서에 기재된'),
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
 'indexing': {'chunk_id': 'chunk_000866',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
