from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조 (보험료의 영수)\n'
 '자동납입일자 또는 급여이체일자는 이 청약서에 기재된 보험료납입 해당일에도 불구하고 회사와 계약자가 별도로 약정한 일자로 합니다.\n'
 '제 3조 (계약 후 알릴 의무)\n'
 '계약자는 지정계좌의 번호가 변경 또는 거래정지된 경우에는 그 사실을 즉시 회사에 알 려야 합니다.\n'
 '제 4조 (준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000631',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
