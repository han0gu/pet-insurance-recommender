from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:14px'>\uf000 회사는 제1조(보험금의 지급사유)에서 정한 "
 '일반상해80%이상후유장해보험금이<br>지급된 때에는 그 지급사유가 발생한 때부터 이 보장은 소멸되며 이 보장의 해약<br>환급금을 '
 '지급하지 않습니다.<br>\uf000 피보험자가 사망하였을 경우에는 이 보장도 소멸되며 회사는 "보험료 및 해약환<br>급금 '
 '산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 보장의 계약자<br>적립액 및 미경과보험료를 계약자에게 '
 "지급합니다.</p><br><p id='198'"),
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
 'indexing': {'chunk_id': 'chunk_000347',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
