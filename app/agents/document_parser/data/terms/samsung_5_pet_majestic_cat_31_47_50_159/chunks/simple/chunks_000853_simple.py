from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 지정대리청구인 변경신청서(회사양식) 2. 지정대리청구인의 가족관계등록부(기본증명서 등) 3. 신분증(주민등록증이나 운전면허증 등 '
 '사진이 붙은 정부기관 발행 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 '
 '수단을 활용한 계약자 의사표시의 확인방법 포함)\n'
 '② 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경 우, 계약자는 이 특별약관에서도 그 기간에 한하여 '
 '지정대리청구인을 변경 지정할 수 있습니다.\n'
 '제5조 (보험금 지급 등의 절차)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000853',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
