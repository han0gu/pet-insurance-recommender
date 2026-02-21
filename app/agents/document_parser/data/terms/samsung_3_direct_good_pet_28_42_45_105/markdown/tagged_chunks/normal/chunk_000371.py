from langchain_core.documents import Document

chunk = Document(
    page_content=('납처분절차에 의해 특별약관이 해지된 경우 해지 당시의 보험수익자가 계약자의 동의\n'
 '를 얻어 특별약관 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제17조\n'
 '(특별약관내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하\n'
 '여 특별약관의 특별부활(효력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합\n'
 '니다.<용어풀이># [강제집행과 담보권실행]강제집행이란 사법상 또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국가가 강제 권력으로'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000371',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
