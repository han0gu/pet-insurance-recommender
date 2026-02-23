from langchain_core.documents import Document

chunk = Document(
    page_content=("id='114' data-category='paragraph' style='font-size:14px'>특<br>제1조(보험금의 "
 '지급사유) 별<br>\uf000 회사는 보험증권에 기재된 반려동물에게 이 특별약관의 보험기간 중 반려동물의료 약<br>비의 '
 '보장개시일(이하 반려동물의료비보장개시일이라 합니다) 이후에 상해 또는 관<br>질병(이하 사고라 합니다)이 발생하여 그 치료를 직접적인 '
 '목적으로 국내에서 수<br>의사에게 치료를 받은 때에는 1일당 피보험자가 부담한 반려동물의 치료에 사용된<br>비용(각종 할인 및 감면, '
 '사후환급금액 등을'),
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
 'indexing': {'chunk_id': 'chunk_000933',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
