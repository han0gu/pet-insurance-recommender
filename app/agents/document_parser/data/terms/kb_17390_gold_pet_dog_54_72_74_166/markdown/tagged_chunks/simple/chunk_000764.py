from langchain_core.documents import Document

chunk = Document(
    page_content=('지 않습니다.제5조(지정대리청구인의 변경지정)계약자는 다음의 서류를제출하고 지정대리청구인을 변경 지정할 수 있습니다. 이 경우회사는 '
 '변경지정을 서면으로 알리거나 보험증권에 그 뜻을 기재하여 드립니다.\n'
 '1. 청구서(회사양식)- 2. 지정대리청구인의 주민등록등본\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인\n'
 '- 특\n'
 '이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰별수단을 활용한 보험수익자 의사표시의 확인방법 포함)# 성이 '
 '확보된 전자적약관\n'
 '제6조(보험금의 청구)'),
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
 'indexing': {'chunk_id': 'chunk_000764',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
