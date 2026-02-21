from langchain_core.documents import Document

chunk = Document(
    page_content=('. 타인의 사망을 보험금 지급사유로 하는 계약에서 계약을 체결할 때까지 피보험<br>자의 서면(전자서명법 제2조 제2호에 따른 전자서명이 '
 '있는 경우로서 상법 시<br>행령 제44조의2에 정하는 바에 따라 본인 확인 및 위조·변조 방지에 대한 신<br>뢰성을 갖춘 전자문서를 '
 '포함)에 의한 동의를 얻지 않은 경우. 다만, 단체가<br>규약에 따라 구성원의 전부 또는 일부를 피보험자로 하는 계약을 체결하는 '
 '경<br>우에는 이를 적용하지 않습니다'),
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
 'indexing': {'chunk_id': 'chunk_000187',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
