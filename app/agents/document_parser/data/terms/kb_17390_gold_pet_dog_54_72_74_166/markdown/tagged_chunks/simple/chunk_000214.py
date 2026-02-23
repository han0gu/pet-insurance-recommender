from langchain_core.documents import Document

chunk = Document(
    page_content=('| 관 련 법 규 신용정보의 이용 및 보호에 관한 법률 ∙ 제32조(개인신용정보의 제공ㆍ활용에 대한 동의) 공 회사가 개인신용정보를 '
 '타인에게 제공하려는 경우 다음 각 호 중 하나의 방식 통 으로 미리 동의를 얻어야 한다. 사항 1. 서면 2. 「전자서명법」 '
 '제2조제2호에 따른 전자서명(서명자의 실지명의를 확인 할 수 있는 것을 말한다)이 있는 전자문서(「전자문서 및 전자거래기본 법」 '
 '제2조제1호에 따른 전자문서를 말한다) 보 3. 유무선통신으로 개인비밀번호를 입력하는 방식 4'),
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
 'indexing': {'chunk_id': 'chunk_000214',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
