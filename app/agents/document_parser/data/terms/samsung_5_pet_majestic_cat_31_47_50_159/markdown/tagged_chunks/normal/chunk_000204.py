from langchain_core.documents import Document

chunk = Document(
    page_content=('에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있습니\n'
 '다.제4관 보험계약의 성립과 유지- 55 -# 제19조 (특별약관의 성립)- ① 이 특별약관은 기본계약(기본계약에 다른 특별약관이 부가된 '
 '경우에는 그 특별약관을\n'
 '- 포함합니다. 이하 같습니다)을 체결할 때 보험계약자의 청약과 보험회사의 승낙으로\n'
 '- 기본계약에 부가하여 이루어집니다\n'
 '- ② 제1항의 규정에도 불구하고 기본계약의 보장개시일 이후에 이 특별약관을 청약하는'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
