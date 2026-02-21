from langchain_core.documents import Document

chunk = Document(
    page_content=('소멸)<br>\uf000 회사는 제1조(보험금의 지급사유)에서 정한 반려동물장례비용지원금을 지급한 경<br>우에는 그 지급사유가 발생한 '
 "때부터 이 특별약관 계약은 소멸되며 이 특별약관</p><br><p id='129' data-category='paragraph' "
 "style='font-size:14px'>의 해약환급금을 지급하지 않습니다.<br>\uf000 보험증권에 기재된 반려동물이 다음 중 "
 '한가지에 해당되는 경우 회사는 "보험료 질<br>및 해약환급금 산출방법서"에서 정하는 바에 따라 반려동물 사망 당시 이 특별약 '
 '병<br>관의 계약자적립액 및'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001125',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
