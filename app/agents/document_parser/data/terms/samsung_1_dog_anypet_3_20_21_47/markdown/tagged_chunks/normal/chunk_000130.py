from langchain_core.documents import Document

chunk = Document(
    page_content=('- 질(원자핵분열 생성물을 포함합니다)의 방사성, 폭발성 또는 그 밖의 유해한 특성에 의한 사고\n'
 '- 8. 위 제7호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '- 9. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 10. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에 대한 태만\n'
 '# 제3조(보험금의 청구)① 피보험자가 반려동물 사망위로금 특별약관 보험금을 청구할 때에는 다음의 서류를 회사에 제출하\n'
 '여야 합니다.- 1. 보험금 청구서(회사 양식)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
