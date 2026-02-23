from langchain_core.documents import Document

chunk = Document(
    page_content=('끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에 나\n'
 '이가 증가하는 것으로 합니다.\n'
 '\uf000 피보험자의 나이 또는 성별에 관한 청약서상 기재사항이73사실과 다른 경우에는 정정된 나이 또는 성별에 해당하는\n'
 '보험금 및 보험료로 변경합니다.# 【 보험나이 계산 예시 】생년월일 : 1988년 10월 2일\n'
 '현재(계약일) : 2023년 4월 14일⇒ 2023년 4월 14일 - 1988년 10월 2일# = 34년 6월 12일 = 35세# 【 '
 '계약해당일 】최초계약일과 동일한 월, 일을 말합니다. 다만, 해당 연'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
