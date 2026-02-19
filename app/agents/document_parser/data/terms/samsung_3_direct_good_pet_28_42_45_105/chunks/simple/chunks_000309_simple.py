from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[현저하게 공정을 잃은 합의]\n'
 '회사가 보험수익자의 경제적․신체적․정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 동일․유사 사례에 비추어 보험수익자에게 매우 '
 '불합리하게 합의를 하는 것을 의미합니다.\n'
 '제44조 (개인정보보호)\n'
 '① 회사는 이 특별약관과 관련된 개인정보를 이 특별약관의 체결, 유지, 보험금 지급 등 을 위하여「개인정보 보호법」,「신용정보의 이용 및 '
 '보호에 관한 법률」등 관계 법 령에 정한 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000309',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
