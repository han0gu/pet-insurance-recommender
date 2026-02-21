from langchain_core.documents import Document

chunk = Document(
    page_content=('가족관계등록부에 기재된 사망연월일을 기준으로 합니다.- \n'
 '<용어풀이>- \n'
 '[실종선고]- \n'
 '어떤 사람의 생사불명의 상태가 일정기간 이상 계속될 때 이해관계가 있는 사람의 청구에 의해 사\n'
 '망한 것으로 간주하는 법원의 결정② 「호스피스·완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따'),
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
 'indexing': {'chunk_id': 'chunk_000271',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
