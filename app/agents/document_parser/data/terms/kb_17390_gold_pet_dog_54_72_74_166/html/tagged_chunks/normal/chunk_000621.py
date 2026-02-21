from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물양육자금Ⅱ(질병사망)</h1><p id='144' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험금의 지급사유)<br>회사는 피보험자가 이 특별약관의 보험기간 중 질병으로 사망한 "
 "경우에는 이 특별약<br>관의 보험가입금액 전액을 반려동물양육자금Ⅱ(질병사망)으로 보험수익자에게 지급</p><br><h1 id='145' "
 "style='font-size:14px'>합니다.</h1><p id='146' data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
