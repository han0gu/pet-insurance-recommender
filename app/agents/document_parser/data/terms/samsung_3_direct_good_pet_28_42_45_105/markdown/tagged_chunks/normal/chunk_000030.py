from langchain_core.documents import Document

chunk = Document(
    page_content=('지난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.# 제11조 (보험수익자의 지정)① 보험수익자를 지정하지 않은 때에는 '
 '보험수익자를 사망보험금의 경우는 피보험자의\n'
 '법정상속인, 기타 보험금의 경우는 피보험자로 합니다.\n'
 '② 제1항에 따라 지정된 보험수익자가 보험기간 중에 사망한 때에는 계약자는 다시 보험\n'
 '수익자를 지정할 수 있으며, 이 경우에 계약자가 보험수익자를 지정하지 않고 사망한\n'
 '때에는 보험수익자의 법정상속인을 보험수익자로 합니다.-'),
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
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
